## Contribution Instructions

To manage our codebase, it is necessary to learn some tools. This may take some time for beginners, but will benefit the rest life. **All the following instructions assume that the reader is familiar and comfortable with linux command line.** 

### Workflow

Steps to contribute your codes to this repository:
1. `git clone` the database to local computer
1. `git pull` to get the latest version
1. Write codes and test locally
1. `git commit` with clean commit message
1. Send pull request with `git push`. **[IMPORTANT]** To avoid messing up the main codebase, please always create a new branch for new changes and avoid directly push to the main branch. Create pull request in github website when you want to merge to the main branch.
1. Repeat 2-5 for next contribution.

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

#### Code Style

We basically follow the [PEP8 standard](https://www.python.org/dev/peps/pep-0008/). Please also refer to the [google style guide](https://google.github.io/styleguide/pyguide.html). Please pay special attention to **indentations, spaces, linebreaks, name style, comments and docstrings**.

#### Anaconda

Please manage your local coding dependencies with [anaconda](https://www.anaconda.com/). 