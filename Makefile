gitbig:
	find . -size +10M
	echo du -a ./ | sort -n -r | head -n 20
	for file in `find . -size +10M`; do ls -lh $$file; done


publish:
	git remote set-url origin git@github.com:tonybutzer/ai-nlcd.git
	#git remote set-url origin https://github.com/tonybutzer/ai-nlcd.git
	git config --global user.email tonybutzer@gmail.com
	git config --global user.name tonybutzer
	git config --global push.default simple
	git add .
	git commit -m "automatic git update from Makefile"
	git push

jup:
	docker run -it -p 80:8888 -v `pwd`:/home/jovyan/ tbutzer/jupyter-rise jupyter notebook --allow-root --ip="0.0.0.0" --NotebookApp.token='yaml'


Image=tbutzer/eco_docker_image

build:
	docker build -t ${Image} .

run:
	docker run -it ${Image} bash

exec:
	docker exec -it ${Image} bash

images:
	docker image ls
	echo ======================================================================
	docker image ls | grep greg

git-hell:
	echo git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch TOO_BIG_FILE'
example:
	echo docker run -i tbutzer/eco_docker_image python3 yr_rangeland.py -y 1993 
