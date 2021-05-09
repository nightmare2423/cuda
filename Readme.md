# Nody程序优化

## nbody.cu
- 基准串行代码
- 0.03billion
## nbody1.cu
- 实现了简单并行，每个线程计算一个点的数据
- 11 billion
## nbody2.cu(最佳版本)
- 每个线程计算一个点的某一部分，用BLOCK_ STEP划分
- 将更新点数据和计算合并到同一个函数中
- 编译优化，for展开
- 56 billion
## nbody2_ 1.cu
- 每个线程计算一个点的某一部分，用BLOCK_ STEP划分
- 编译优化，for展开
- 51 billion
## nbody2_ 2.cu
- 每个线程计算一个点的某一部分，用BLOCK_ STEP划分
- 编译优化，for展开
- 考虑到硬件特点，一定是最后一组最后算完，故在最后一组顺便计算结果
- 55 billion
