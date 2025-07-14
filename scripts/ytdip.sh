

#!/bin/bash
# yt_to_xtts.sh
# Usage: ./yt_to_xtts.sh "<YouTube URL>" <output_name>
# Example: ./yt_to_xtts.sh "https://www.youtube.com/watch?v=3xU3wVo5vWI" speaker1

# Prerequisites:
# - Install yt-dlp: brew install yt-dlp
# - Install ffmpeg: brew install ffmpeg

URL="$1"
OUTNAME="$2"

if [[ -z "$URL" || -z "$OUTNAME" ]]; then
  echo "Usage: ./yt_to_xtts.sh \"<YouTube URL>\" <output_name>"
  exit 1
fi

mkdir -p reference_voices

echo "Downloading and extracting voice clip for $OUTNAME..."
yt-dlp --quiet --no-warnings --download-sections "*00:03:05-00:03./y:20" -x --audio-format wav -o "temp.%(ext)s" "$URL"

ffmpeg -i temp.wav -ss 00:00:00 -t 00:08 -ar 16000 -ac 1 "reference_voices/$OUTNAME.wav" -y
rm -f temp.wav

echo "✅ Saved: reference_voices/$OUTNAME.wav"