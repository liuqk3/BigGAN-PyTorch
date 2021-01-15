cd TFHub
export CUDA_VISIBLE_DEVICES="1,2,3" && python converter.py --resolution 256 --redownload --verbose --generate_samples --batch_size 32 --paralle # --only_sample
cd ..