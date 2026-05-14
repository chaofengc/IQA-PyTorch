## Contribution Instructions

### Add a New IQA Model (Checklist)

When adding a new model, please follow existing implementations and complete all items below before opening a PR:

1. **Add model architecture file** in `pyiqa/archs/xx_arch.py`.
	- Register the model class with `@ARCH_REGISTRY.register()`.
	- Keep initialization, preprocessing, and forward logic aligned with existing models.
	- If pretrained weights are needed, follow current practice in `pyiqa/archs/arch_util.py` (Hugging Face weight URL + `load_pretrained_network`).

1. **Add default model config** in `pyiqa/default_model_configs.py`.
	- Add a new entry in `DEFAULT_CONFIGS` with a unique model name.
	- Define at least `metric_opts` and `metric_mode` (`FR` or `NR`).
	- Add `lower_better` and `score_range` when applicable.

1. **Upload weights to Hugging Face**.
	- Upload model checkpoints to: `https://huggingface.co/chaofengc/IQA-PyTorch-Weights`.
	- Use stable file naming so it is easy to map config/model option to checkpoint file.
	- Ensure model code can download/load the uploaded checkpoints directly.

1. **Update model card** in `docs/ModelCard.md`.
	- Add the new model name, method type (FR/NR/task-specific), short description, and score direction.
	- If there are known constraints (input size, no backward, special preprocessing), document them clearly.

1. **Update README** in `README.md`.
	- Add the model to the corresponding method list or changelog section when needed.
	- Keep README and ModelCard model names consistent with `list_models()` output.

1. **Run tests and open PR**.
	- Verify the model can be created and inferred correctly (for example, via `pyiqa -ls`, `pyiqa.create_metric(...)`, or `python inference_iqa.py ...`).
	- Run relevant benchmark/training scripts if your change affects them.
	- Submit a clean PR with concise description, key results, and related paper/repo links.

### 新增 IQA 模型（清单）

当你在本仓库新增模型时，请参考已有实现，并在提交 PR 前完成以下事项：

1. **新增模型结构文件**到 `pyiqa/archs/xx_arch.py`。
	- 使用 `@ARCH_REGISTRY.register()` 注册模型类。
	- 初始化、预处理和 forward 逻辑尽量与现有模型风格保持一致。
	- 如需预训练权重，请遵循 `pyiqa/archs/arch_util.py` 的现有方式（Hugging Face 权重 URL + `load_pretrained_network`）。

1. **在** `pyiqa/default_model_configs.py` **中新增默认模型配置**。
	- 在 `DEFAULT_CONFIGS` 中新增条目，并使用唯一模型名。
	- 至少定义 `metric_opts` 与 `metric_mode`（`FR` 或 `NR`）。
	- 根据需要补充 `lower_better` 与 `score_range`。

1. **上传权重到 Hugging Face**。
	- 将模型权重上传到：`https://huggingface.co/chaofengc/IQA-PyTorch-Weights`。
	- 使用稳定且清晰的文件命名，便于从配置/模型选项映射到权重文件。
	- 确保模型代码可以直接下载并加载上传后的权重。

1. **更新** `docs/ModelCard.md` **中的模型卡**。
	- 增加新模型名称、方法类型（FR/NR/任务专项）、简要描述和分数方向。
	- 若有已知限制（输入尺寸、不可反传、特殊预处理），请明确写出。

1. **更新** `README.md`。
	- 按需将新模型加入对应的方法列表或更新日志部分。
	- 保证 README 与 ModelCard 中的模型名和 `list_models()` 输出一致。

1. **测试通过后提交 PR**。
	- 验证模型可正常创建并推理（例如使用 `pyiqa -ls`、`pyiqa.create_metric(...)` 或 `python inference_iqa.py ...`）。
	- 如果改动会影响基准或训练流程，请运行对应 benchmark/training 脚本。
	- 提交整洁的 PR，附上简要说明、关键结果与相关论文/仓库链接。


To manage our codebase, it is necessary to learn some tools. This may take some time for beginners, but will benefit the rest life. **All the following instructions assume that the reader is familiar and comfortable with linux command line.**

### Workflow

Steps to contribute your codes to this repository:
1. Fork the repo to your own github account.
1. `git clone` and `git pull` the forked repo to your own computer.
1. Create your own branch `git checkout -b`. Write codes and test.
1. `git commit` your changes with clean commit message.
1. Push your branch to your forked repo with `git push`.
1. Head on over to the forked repo on GitHub, and open a pull request to merge your changes to main project. 
1. Get your PR reviewed and approved.

### Style

- **Code style.** We basically follow the [PEP8 standard](https://www.python.org/dev/peps/pep-0008/). Please also refer to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Please pay special attention to **indentations, spaces, linebreaks, name style, comments and docstrings**.
- **Doc style.** Please refer to [Google Python Style Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- **Commit message.** Please refer to [Git Commit Message Conventions](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#) for good examples of commit message. You may also use tools like [cz-cli](https://github.com/commitizen/cz-cli).

### Tools and Codes

#### VSCode

[VSCode](https://code.visualstudio.com/) is the recommended IDE for coding. It is extremely powerful with the following plugins:
- remote ssh: link to remote server
- debug python codes by simply adding breakpoints with mouse click
- autoformat your code to pep8 standard
- nested jupyter notebook
- markdown editing and preview
- and so on...

#### Github

Please always use git to track your codes. Refer to the [Pro Git book](https://git-scm.com/book/en/v2) for comprehensive understanding of git. You may also get a quick start with the [git cheatsheet](https://jan-krueger.net/wordpress/wp-content/uploads/2007/09/git-cheat-sheet.pdf). Also, the chinese [Git教程](https://www.liaoxuefeng.com/wiki/896043488029600) is also available. Some suggestions:
- Write clean commit message when you push to this repository.
- If conflict happens when you push your code, you can pull down the repository first with `git pull origin main` and fix the merge.



#### Anaconda

Please manage your local coding dependencies with [anaconda](https://www.anaconda.com/).
