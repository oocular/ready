# LabelStudio
> Open Source Data Labeling Platform [https://labelstud.io/](https://labelstud.io/)

## Setting up LabelStudio and Data Paths

1. Clone the udetect repository and set labelstudio as currenty directory:
```bash
cd $HOME/repositories/oocular/ready/docs/labelstudio
```

2. Add the data you want to label to a subdirectory within the /myfiles folder, e.g. /myfiles/ready:

3. Create my data path with root permisions

```bash
mkdir -p mydata
chmod -R 777 mydata #Never Use chmod 777
#Setting 777 permissions (chmod 777) to a file or directory means that it will be readable, writable and executable by all users and may pose a huge security risk. [https://linuxize.com/post/what-does-chmod-777-mean/]
```



