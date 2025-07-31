# test_sim.py

## 测试环境设置

为了运行这些测试，你需要安装必要的依赖：

```bash
pip install pytest pyyaml requests graphviz tqdm
```

## 运行测试

在项目根目录下运行以下命令来执行测试：

```bash
pytest test/test_sim.py -v
```

这些测试用例覆盖了代码中的主要功能点，包括：

- 配置文件加载
- 比赛模拟逻辑
- 积分计算
- 排名确定
- 季后赛流程

通过这些测试，可以确保代码的核心功能正常工作，并且在后续修改时能够及时发现问题。
