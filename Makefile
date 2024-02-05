default: vast_install


help:
	@echo "HOST=\$VAST_HOST PORT=\$VAST_PORT make setup_remote"
	@echo "scp -P \$VAST_PORT -r test_script.py root@\$VAST_HOST:/root/obvs"
	@echo "ssh -p \$VAST_PORT root@\$VAST_HOST"
	@echo "cd obvs && make vast_install"
	@echo "if you need a huggingface model: poetry run huggingface-cli login"
	@echo "poetry run python test_script.py"



setup_remote:
	ssh -p $(PORT) root@$(HOST) "git clone https://github.com/fergusFettes/obvs && cd obvs && git checkout llama2_patchscopes_nnsight && apt-get install make"


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

