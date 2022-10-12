import os
import re
import urllib.request
from pytube import YouTube
import argparse
from rich import print as rprint
from rich.progress import track
from pydub import AudioSegment


class YouTubeScrapper:

    def __init__(self, only_audio: bool = True, output_folder: str = "Music"):
        self.only_audio = only_audio
        self.output_folder = output_folder
        self.file_name = None

    def download(self, song_name):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.get_video(song_name)

    @staticmethod
    def song_name_preprcessor(song_name):
        return song_name.replace(" ", "+")

    def get_video_link(self, song_name : str) -> str:
        print(f"Now Downloading ... {song_name}.mp3")
        song_name = self.song_name_preprcessor(song_name)
        html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + song_name)
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        return "https://www.youtube.com/watch?v=" + video_ids[0]

    def get_video(self, song_name):
        out_file = YouTube(self.get_video_link(song_name)).streams.filter(only_audio=True).first().download(
            os.getcwd() + '/' + self.output_folder)
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)
        self.file_name = new_file
        return self.cut_audio(0, 30)

    def cut_audio(self, start_time, end_time):
        if self.file_name is None:
            rprint("[bold red]No file found[/bold red]")
            return
        start_time = start_time * 1000
        end_time = end_time * 1000
        new_audio = AudioSegment.from_mp3(self.file_name)
        new_audio = new_audio[start_time:end_time]
        new_audio.export(self.file_name, format="mp3")
        return self.file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download youtube audio ")
    parser.add_argument("-s", "--songs", help="list of songs to download seperated with comma(,) add songs",
                        required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=False, default="Music")
    args = parser.parse_args()
    yt = YouTubeScrapper(output_folder=args.output)
    for song in track(args.songs.split(","), description="Downloading songs..."):
        # for song in args.songs.split( "," ):
        yt.get_video(song)
    rprint("Downloaded", args.songs)
    # yt.get_video(args.songs)
