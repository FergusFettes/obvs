setup_remote:
	ssh -p $(PORT) root@$(HOST) "git clone https://github.com/fergusFettes/obvs && cd obvs && git checkout llama2_patchscopes_nnsight && apt-get install make"

all:
	make vast_install

vast_install:
	make add_deadsnakes
	make apt_installs
	make python_installs

add_deadsnakes:
	# For some reason you just need to run this twice and it works?
	add-apt-repository ppa:deadsnakes/ppa

apt_installs:
	apt-get update && apt-get upgrade -y
	DEBIAN_FRONTEND=noninteractive apt-get install python3.11 neovim -y

python_installs:
	pip install poetry
	poetry install

watch:
	watch -n 0.1 -d "nvidia-smi"

