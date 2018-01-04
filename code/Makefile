all:
	sudo python setup.py install
	sudo python setup.py develop
	
dist:
	# Remove previous packages
	rm -rf dist
	# Pack the current version
	python setup.py dist
