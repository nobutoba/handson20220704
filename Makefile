.PHONY : dev
dev :
	rm -rf docs/build/
	sphinx-apidoc  -f -o ./docs/source ./my_package
	sphinx-autobuild -b html --watch my_package/ docs/source/ docs/build/

.PHONY : build
build :
	rm -rf docs/build/
	sphinx-apidoc  -f -o ./docs/source ./my_package
	sphinx-build ./docs/source ./docs/build

.PHONY : clean
clean :
	rm -rf docs/build/
