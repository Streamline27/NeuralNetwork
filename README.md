
+# NeuralNetwork
+
+Для того, чтоб скомпилировать и обеспечить работу программы, необходимо выполнить следующие шаги:
+
+1. Скачайте архив **ToWorkingDir** по ссылке: http://www.filedropper.com/toworkingdir
+2. Скопировать содержимое папки *toWorkingDir* (Не считая каталогов *libx32* и *libx64*) и поместить в одну папку с .exe файлом.
+3. Извлечь содержимое папки *libx32* или *libx64* в папку с .exe файлом (Выбор каталога зависит от разрядности вышей системы).
+4. Скачать библиотеку **Armadillo** по ссылке http://arma.sourceforge.net/download.html.
+5. Указать путь на *include* каталог скачаной библиотеки Armadillo в C/C++>Additional Include Directories.
+6. Указать путь к каталогу *NeuralNetwork/algib* в C/C++>Additional Include Directories.
+7. Указать каталог содержащий **libopenblas.lib** в General>Additional Library Directories. (Входит в папку *libx32* или *libx64*).
+8. Прописать строку *libopenblas.lib* в Input>Additional Dependencies
