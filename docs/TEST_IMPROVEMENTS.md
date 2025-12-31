# 防回归测试改进总结

## 问题分析

### 原始测试的问题

1. **无法证明 force_execute 阻止了 column pruning**
   - 使用 `F.lit(random.random())` 被固化成常量
   - 测试在 `force_execute` 后又做了 `agg(sum(z))`，这个 action 本身就会强制计算 z
   - 执行计划断言太弱

2. **无法证明 force_execute 的必要性**
   - 后续的 action 足以保证列被计算

3. **无法证明 Ray 不调用 materialize**
   - 没有任何机制检测 materialize() 是否被调用
   - `take_all()` 本身就会触发执行

## 改进方案

### 1. force_execute 返回 checksum

**修改**: `force_execute()` 现在返回 `float` 类型的 checksum 值

**优势**:
- 可以通过比较不同变换的 checksum 验证是否依赖输出列值
- 测试可以真正验证 force_execute 的行为

**实现**:
```python
def force_execute(obj: Any, *, engine: str, cols: Sequence[str]) -> float:
    """返回 checksum 值（用于验证是否依赖输出列）"""
    ...
```

### 2. Spark 测试改进

#### `test_spark_force_execute_depends_on_output_values`

**改进前**:
- 使用 `F.lit(random.random())`（常量）
- 在 force_execute 后又做 `agg(sum(z))`
- 无法证明 force_execute 依赖输出列值

**改进后**:
```python
df2 = df.withColumn("z", F.col("x") * 2.0)
df3 = df.withColumn("z", F.col("x") * 3.0)

chk2 = force_execute(df2, engine="spark", cols=["z"])
chk3 = force_execute(df3, engine="spark", cols=["z"])

# 关键断言：checksum 必须不同
assert chk2 != chk3, "force_execute 必须依赖输出列 z 的值"

# 对照：count 不依赖列值，两者应相同
assert df2.count() == df3.count() == 3
```

**验证点**:
- ✅ 不同变换得到不同 checksum → 证明依赖输出列值
- ✅ count() 对两种变换结果相同 → 证明 count 不依赖列值

#### `test_spark_force_execute_with_multiple_output_cols`

**改进**:
```python
df_a = df.withColumn("z1", x*2).withColumn("z2", y*3)
df_b = df.withColumn("z1", x*2).withColumn("z2", y*4)

chk_a = force_execute(df_a, engine="spark", cols=["z1", "z2"])
chk_b = force_execute(df_b, engine="spark", cols=["z1", "z2"])

assert chk_a != chk_b  # 证明依赖输出列值
```

### 3. Ray 测试改进

#### `test_ray_force_execute_does_not_call_materialize`

**改进**: 使用 `pytest.monkeypatch` 确保 materialize() 不被调用

```python
def test_ray_force_execute_does_not_call_materialize(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("force_execute 不应调用 Dataset.materialize()")
    
    monkeypatch.setattr(Dataset, "materialize", _boom, raising=True)
    
    # 只要这里不抛 boom，就证明没调用 materialize
    _ = force_execute(ds2, engine="ray", cols=["z"])
```

**验证点**:
- ✅ 如果调用 materialize() 会报错
- ✅ 测试通过 → 证明没调用 materialize

#### `test_ray_force_execute_depends_on_output_values`

**新增测试**: 验证 Ray force_execute 依赖输出列值

```python
ds2 = ds.map_batches(lambda df: df.assign(z=df["x"]*2), ...)
ds3 = ds.map_batches(lambda df: df.assign(z=df["x"]*3), ...)

chk2 = force_execute(ds2, engine="ray", cols=["z"])
chk3 = force_execute(ds3, engine="ray", cols=["z"])

assert chk2 != chk3  # 证明依赖输出列值
```

### 4. 资源释放改进

**改进**: 所有测试使用 `try/finally` 确保资源释放

```python
spark = get_spark("Test")
try:
    # 测试代码
finally:
    spark.stop()  # 确保即使断言失败也释放资源
```

## 测试覆盖

### ✅ 现在能真正验证的点

1. **Spark column pruning 防护**
   - ✅ 不同变换得到不同 checksum
   - ✅ 对照：count() 不依赖列值

2. **Ray materialize 避免**
   - ✅ monkeypatch 确保不调用 materialize()
   - ✅ 不同变换得到不同 checksum

3. **多列处理**
   - ✅ 多个输出列的 checksum 正确

4. **Fallback 行为**
   - ✅ 空 cols 时 fallback 到 count()

## 关键改进点

### 1. 返回值设计
- `force_execute()` 返回 checksum
- 测试通过比较 checksum 验证行为
- 实际使用可以忽略返回值

### 2. 测试策略
- **不依赖后续 action**: 测试只验证 force_execute 本身
- **可区分验证**: 不同变换得到不同 checksum
- **对照测试**: count() 不依赖列值

### 3. 资源管理
- 所有测试使用 try/finally
- 确保资源正确释放

## 文件变更

### 修改的文件
1. **`src/bench/materialize.py`**
   - `force_execute()` 返回 `float` (checksum)
   - `_force_execute_spark()` 返回 checksum
   - `_force_execute_ray()` 返回 checksum

2. **`tests/test_benchmark_trigger.py`**
   - 完全重写，使用 checksum 验证
   - 添加 monkeypatch 测试
   - 改进资源释放

## 验证

运行测试：
```bash
python -m pytest tests/test_benchmark_trigger.py -v
```

预期结果：
- ✅ `test_spark_force_execute_depends_on_output_values` - 验证依赖输出列值
- ✅ `test_spark_force_execute_with_multiple_output_cols` - 验证多列处理
- ✅ `test_ray_force_execute_does_not_call_materialize` - 验证不调用 materialize
- ✅ `test_ray_force_execute_depends_on_output_values` - 验证依赖输出列值
- ✅ `test_force_execute_fallback_to_count` - 验证 fallback 行为

