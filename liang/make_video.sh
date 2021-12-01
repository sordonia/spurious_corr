
# first arg is the directory with png stuff

ffmpeg -framerate 8 -i $1/boundary_e%d.png -r 30 -pix_fmt yuv420p $1/video.mp4