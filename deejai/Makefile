build-DeejaiFunction:
	cp *.py $(ARTIFACTS_DIR)
	cp requirements.txt $(ARTIFACTS_DIR)
	eval "python -m pip install -r requirements.txt -t $(ARTIFACTS_DIR)"
	cp tflite_runtime-2.1.2-cp37-cp37m-linux_x86_64.whl $(ARTIFACTS_DIR)
	eval "python -m pip install tflite_runtime-2.1.2-cp37-cp37m-linux_x86_64.whl -t $(ARTIFACTS_DIR)"
	rm -rf $(ARTIFACTS_DIR)/bin
	rm $(ARTIFACTS_DIR)/tflite_runtime-2.1.2-cp37-cp37m-linux_x86_64.whl
