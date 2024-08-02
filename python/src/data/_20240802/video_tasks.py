import os
from typing import Optional, List

from pytasuku import Workspace


class VideoTasksArgs:
    def __init__(self,
                 prefix: str,
                 frame_file_pattern: str,
                 num_frames: int,
                 dependencies: Optional[List[str]] = None):
        if dependencies is None:
            self.dependencies = []

        self.dependencies = dependencies
        self.num_frames = num_frames
        self.frame_file_pattern = frame_file_pattern
        self.prefix = prefix

    def video_file_name(self):
        return f"{self.prefix}/video.mp4"

    def video_for_web_file_name(self):
        return f"{self.prefix}/video_for_web.mp4"

    def create_video(self):
        os.makedirs(self.prefix, exist_ok=True)
        command = "ffmpeg " \
                  + "-y " \
                  + "-framerate 30 " \
                  + "-i " + self.frame_file_pattern + " " \
                  + "-c:v libx264rgb " \
                  + "-crf 0 " \
                  + "-r 30 " \
                  + self.video_file_name()
        os.system(command)

    def create_video_for_web(self):
        os.makedirs(self.prefix, exist_ok=True)
        command = "ffmpeg " \
                  + "-y " \
                  + "-i " + self.video_file_name() + " " \
                  + "-vcodec libx264 -pix_fmt yuv420p -acodec aac " \
                  + "-strict -2 -ac 2 -ab 160k -preset slow -f mp4 " \
                  + self.video_for_web_file_name()
        os.system(command)

    def all_command_name(self):
        return f"{self.prefix}/all"

    def define_tasks(self, workspace: Workspace):
        workspace.create_file_task(
            self.video_file_name(),
            self.dependencies,
            self.create_video)

        workspace.create_file_task(
            self.video_for_web_file_name(),
            [self.video_file_name()],
            self.create_video_for_web)

        workspace.create_command_task(
            self.all_command_name(),
            [
                self.video_file_name(),
                self.video_for_web_file_name()
            ])
