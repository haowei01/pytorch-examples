mkdir -p ckptdir
# Download MSLR-WEB10K
echo "Download MSLR-WEB10K"
mkdir -p data/mslr-web10k/
## TODO: download data and move into this directory
unzip MSLR-WEB10K.zip 'Fold1/*' -d data/mslr-web10k/


# Download Expedia Data
mkdir -p data/expedia/
echo "Downloading Expedia Data"
pushd data/expedia/
kaggle competitions download -c expedia-personalized-sort
unzip data.zip
zip -r -X test.zip test.csv
zip -r -X train.zip train.csv
rm test.csv train.csv data.zip
popd