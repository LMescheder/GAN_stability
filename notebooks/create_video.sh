EXEC=ffmpeg
declare -a FOLDERS=(
	"simgd"
	"altgd1"
       	"altgd5"
)
declare -a SUBFOLDERS=(
	"gan" 
	"gan_consensus" 
	"gan_gradpen" 
        "gan_gradpen_critical" 
        "gan_instnoise" 
	"nsgan"  
	"nsgan_gradpen" 
	"wgan"  
       	"wgan_gp" 
)
OPTIONS="-y"

cd ./out
for FOLDER in ${FOLDERS[@]}; do
	for SUBFOLDER in ${SUBFOLDERS[@]}; do
		INPUT="$FOLDER/animations/$SUBFOLDER/%06d.png"
		OUTPUT="$FOLDER/animations/$SUBFOLDER.mp4"
		$EXEC -framerate 30 -i $INPUT $OPTIONS $OUTPUT
		echo $FOLDER
		echo $SUBFOLDER
	done

done
