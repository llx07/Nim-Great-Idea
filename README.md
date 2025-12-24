# 伟大思想小组项目 - Nim 游戏

组员：林乐逍，唐哲宇，靳泊楠。


## 项目介绍

本项目是 ACM 班 2025 级伟大思想课程项目 AI 方向 Nim 游戏。主要以强化学习为核心，同时实现了一些传统的博弈算法。

本项目主要实现了：
1. Nim 游戏的环境类。
2. 运用多种算法实现的智能体，包括：
    - Q Learning
    - DQN
    - Minimax 搜索
    - 数学最优策略
3. 对多种策略进行胜率的评估，并以表格的形式输出。



## 项目架构

- `core/nim.py`：游戏环境
- `model/*`：DQN 和 Q Learning 的模型文件
- `agents/*`：各类智能体的具体实现。
- `main.py`：项目入口点，支持不同智能体之间博弈、并输出胜率。

## 参考结果

|对方\己方胜率\己方|随机策略|Q Learning|DQN|Alpha-Beta 剪枝|数学最优|
|---|---|---|---|---|---|
|对方是 随机策略| \ |76.4%|48.5%|100.0%|99.8%|
|对方是 Q Learning|23.6%| \ |50.6%|99.8%|99.8%|
|对方是 DQN|51.5%|49.4%| \ |98.8%|99.0%|
|对方是 Alpha-Beta 剪枝|0.0%|0.2%|1.2%| \ |49.7%|
|对方是 数学最优|0.2%|0.2%|1.0%|50.3%| \ |



## 安装


1. 安装 python 虚拟环境：
    ```sh
    sudo apt update
    sudo apt install python3-venv
    ```


1. 克隆仓库

    ```sh
    git clone https://github.com/llx07/Nim-Great-Idea.git
    ```

1. 进入仓库目录后，创建虚拟环境并安装依赖
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```


## 运行

首先激活虚拟环境（在 VS Code 中会自动激活）：
```sh
source .venv/bin/activate
```

然后用命令 `python3` 运行。
```sh
python3 main.py
```

如果要训练 AI，请单独运行 AI 对应的 agent 文件即可进行训练，结果会自动保存在 model 中。

## 开发与协作

1. 创建功能分支

    `main` 分支有写保护，所以你不应该在 `main` 分支上创建任何 commit。

    所有功能开发需在功能分支上进行。

    创建一个分支并切换：

    ```sh
    git checkout -b feat/xxx
    ```

1. 定期从 `main` 分支拉取更改

    通过变基操作将当前功能分支变基到 `main` 的最新更改上（可能需要解决冲突）：

    ```sh
    git fetch origin
    git rebase origin/main
    ```


1. 提交 Pull Request

    为了向 `main` 分支合并你的代码，需要提交一个 Pull Request（PR）。

    在提交 PR 之前：

    - 确保同步了 `main` 分支的最新代码：
        ```sh
        git pull origin main --rebase
        ```    

    - （可选）使用交互式 rebase 整理你的提交记录（[参考资料](https://www.jianshu.com/p/c1e9590c520f)）：
        ```sh
        git rebase -i origin/main
        ```

    - 已经将你的功能分支推到了云端仓库：
        ```sh
        git push --set-upstream origin feat/xxx
        ```

    然后打开仓库 <https://github.com/llx07/Nim-Great-Idea>，就能用网页发起 Pull Request 了。

    之后等到 PR 通过之后，代码会合并到 `main` 分支，再使用 `git pull` 合并到本地:
    ```sh
    git checkout main
    git pull origin main
    ```

1. 后续清理

    可以删除无用的功能分支：
    ```
    git branch -d feat/xxx
    git push origin -d feat/xxx
    ```
