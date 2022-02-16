DOCKERFILE=Dockerfile
# TENSORFLOW_VERSION=2.4.0-gpu
TENSORFLOW_VERSION=2.7.0-gpu
CONTAINER_NAME=tta_cifar10
IMAGE_NAME=ishihara/tta_cifar10:$(TENSORFLOW_VERSION)
COMPOSE_FILE=docker-compose.yml # change this if you have another custom compose file

# paths
make_dir=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))
project_root_dir=$(abspath $(make_dir)../)
working_dir=/root/$(notdir $(abspath $(make_dir)../))
# bashrc=$(project_root_dir)/etc/custom.bashrc

# compose args
COMPOSE_ARGS=TENSORFLOW_VERSION=$(TENSORFLOW_VERSION) WORK_DIR=$(working_dir) \
		CONTAINER_NAME=$(CONTAINER_NAME) IMAGE_NAME=$(IMAGE_NAME) \
		HOST_DIR=$(project_root_dir)
# BASHRC=$(bashrc)


.PHONY: setid init build up bash down help
.DEFAULT_GOAL :=  help


setid: ## set container id
	$(eval CONTAINER := $(shell docker ps -aqf 'name=$(CONTAINER_NAME)' ))

init: build up ## docker-compose build && docker-compose up -d

reset: down up bash

build: ## docker-compose build
#	docker-compose build --build-arg tensorflow_version=$(TENSORFLOW_VERSION) --build-arg work_dir=/home
	@$(COMPOSE_ARGS) docker-compose -f $(COMPOSE_FILE) build

up: setid ## docker-compose up -d
	@$(COMPOSE_ARGS) docker-compose -f $(COMPOSE_FILE) up -d

bash: setid ## docker exec -it CONTAINER /bin/bash
	@docker exec -it $(CONTAINER_NAME) bash

down: setid ## docker stop && rm CONTAINER
	@docker stop $(CONTAINER)
	@docker rm $(CONTAINER)

gpu: setid ## check gpu
	@docker exec $(CONTAINER) $(working_dir)/share/gpu_check.py
	@docker exec $(CONTAINER) $(working_dir)/share/mnist.py

rmi: ## remove all <none> tag images
	@docker rmi $(shell docker images -f "dangling=true" -q)

help: ## show this help
	@echo "make subcommand list:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo working directory map:
	@echo " - $(project_root_dir):$(working_dir)"
