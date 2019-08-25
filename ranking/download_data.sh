mkdir -p ckptdir
mkdir -p data/expedia/

# Download Expedia Data
echo "Downloading Expedia Data"
pushd data/expedia/
kaggle competitions download -c expedia-personalized-sort
unzip data.zip
zip -r -X test.zip test.csv
zip -r -X train.zip train.csv
rm test.csv train.csv data.zip
popd