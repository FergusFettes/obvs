# You will need to 
# git clone https://github.com/fergusFettes/obvs
# cd obvs && git checkout multitoken_remote
vast_install:
	pip install poetry
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt-get update && sudo apt-get upgrade -y
	sudo apt-get install python3.11 neovim -y
	poetry install
