## Contribution Instructions

To manage our codebase, it is necessary to learn some tools. This may take some time for beginners, but will benefit the rest life. **All the following instructions assume that the reader is familiar and comfortable with linux command line.**

### Workflow

Steps to contribute your codes to this repository:
1. `git clone` and `git pull` the latest codes
1. Create your own branch `git checkout -b`. Write codes and test.
1. `git commit` your changes with clean commit message
1. Push your branch to github with `git push origin your_branch`.
1. Open a pull request to merge your branch
1. Get your PR reviewed and approved

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
