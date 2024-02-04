# You will need to 
# git clone https://github.com/fergusFettes/obvs
# cd obvs && git checkout multitoken_remote
all:
	make vast_install

vast_install:
	sudo add-apt-repository ppa:deadsnakes/ppa
	make apt_installs
	make python_installs

apt_installs:
	sudo apt-get update && sudo apt-get upgrade -y
	DEBIAN_FRONTEND=noninteractive sudo apt-get install python3.11 neovim -y

python_installs:
	pip install poetry
	poetry install

watch:
	watch -n 0.1 -d "nvidia-smi"
