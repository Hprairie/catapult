# Sphinx Instruction

1. Compile using `make html` on the cuda server. Couldn't get it to work well on mac
2. Run `sphinx-apidoc -o docs/source ../your_package_name` to generate rst ffiles for each module 
3. rst files are basically the config files for Sphinx to build the docs off of
4. Update index.rst to include these new modules

just append the name of the rst at the bottom. E.g -->

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   new-module

5. Build using `make clean` then `make html` to build htmls. Run `open _build/html/index.html` to see your docs!


# Breathe Instruction   