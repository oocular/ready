
# create and change path
mkdir -p $HOME/datasets/ready/facelandmarks && cd $HOME/datasets/ready/facelandmarks

# download
wget https://github.com/davisking/dlib-models/raw/refs/heads/master/shape_predictor_68_face_landmarks.dat.bz2 -O shape_predictor_68_face_landmarks.dat.bz2
wget https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/refs/heads/master/shape_predictor_81_face_landmarks.dat -O shape_predictor_81_face_landmarks.dat

# decompress and remove
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
rm shape_predictor_68_face_landmarks.dat.bz2
