#!/usr/bin/env python3

import argparse
import os
import hashlib
import sys
import subprocess as sp
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from skimage.io import imshow
from pyzbar import pyzbar

def compute_ssim(frame_s, frame_p):
    if np.array_equal(frame_s, frame_p):
        return 1.0
    return structural_similarity(frame_s, frame_p, channel_axis=2)

# Taken from https://stackoverflow.com/a/55542529
def compute_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()

def files_equal(frame_s, frame_p):
    return compute_hash(frame_s) == compute_hash(frame_p)

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
Y_POS = 56
# X_POS = 1920 - 896
X_POS = 512
PXSIZE = 16

def open_video(path):
    cmd = ['ffmpeg', '-i', path, '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-']
    pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
    return pipe

def parse_timestamp_timecode(frame, lineoffset):
    y_offset = Y_POS + lineoffset*PXSIZE
    x_offset = X_POS
    ts = 0
    for bit in range(0, 64):
        y = y_offset + int(PXSIZE/2)
        x = x_offset + bit*PXSIZE + int(PXSIZE/2)
        rgb = frame[y,x]

        if rgb.mean() >= 200:
            ts |= 1 << (63 - bit)
        elif not rgb.mean() <= 50:
            return -1
    return ts

def parse_timestamp_qr(frame, lineoffset):
    lens = frame[40:250, 40:250]
    codes = pyzbar.decode(lens, symbols=[pyzbar.ZBarSymbol.QRCODE])
    if len(codes) == 0:
        return -1
    return int(codes[0].data)


def show_frame(frame):
    imshow(frame)
    plt.show()

def get_frame(pipe):
    frame_raw = pipe.stdout.read(VIDEO_WIDTH * VIDEO_HEIGHT * 3)
    frame = np.frombuffer(frame_raw, dtype='uint8')
    if frame.shape[0] == 0:
        return None
    else:
        frame = frame.reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3))
    return frame

def get_writer(path, append=False):
    if path is None:
        return lambda v: None
    f = open(path, 'a' if append else 'w')
    if not append:
        f.write("ix_s\tix_p\tssim\tmsg\n")
        f.flush()
    def write(ix_s, ix_p, ssim, msg):
        if ix_s is None:
            f.close()
        else:
            m = msg if msg is not None else ''
            f.write(f'{ix_s}\t{ix_p}\t{ssim:.6f}\t{m}\n')
            f.flush()
    return write

def get_csv_path(path):
    dir = os.path.dirname(path)
    name, ext = os.path.splitext(os.path.basename(path))
    return os.path.join(dir, f"{name}.ssim.csv")


# Adapted from https://stackoverflow.com/a/68199266
def main():
    parser = argparse.ArgumentParser(description="Compute the SSIM of two videos")
    parser.add_argument('pi', help='The source video')
    parser.add_argument('server', help='The recorded video')
    parser.add_argument('--save', help='Save output as csv', action='store_true', default=False)
    parser.add_argument('--start', help='Start processing at this frame nr', type=int, default=0)
    parser.add_argument('--stop', help='Stop processing at this frame nr', type=int, default=0)
    parser.add_argument('--method', help='Method to read timestamps', choices=['qr', 'timecode'], default='qr')
    args = parser.parse_args()

    # Export the frames using ffmpeg
    # ffmpeg -i server.mkv -copyts -vsync 0 -f image2 -frame_pts true 'frames_server/%05d.jpg'

    pipe_s = open_video(args.pi)
    pipe_p = open_video(args.server)

    parse_timestamp = parse_timestamp_qr if args.method == 'qr' else parse_timestamp_timecode

    need_newline = False
    def get_prefix():
        nonlocal need_newline
        return f"[{ts_s:05d}|{ts_p:05d}]"

    def log(string):
        nonlocal need_newline
        print(('\n' if need_newline else '') + get_prefix() + ' ' + string)
        need_newline = False

    start_up = True
    frame_s = None
    frame_p = None
    ts_s = 0
    ts_p = 0
    fetch_s = True
    fetch_p = True
    writer = get_writer(get_csv_path(args.server), append=(args.start > 0))

    while True:
        prev_ts_s = 0
        if fetch_s:
            frame_s = get_frame(pipe_s)
            if frame_s is None:
                log(f"End video streamer")
                break
            prev_ts_s = ts_s
            ts_s = parse_timestamp(frame_s, 7)
            if ts_s < 0:
                log(f"Failed to parse ts from frame_s")
                break
            if ts_s < args.start:
                log("Skip forward in video 1 ...")
                continue
            # if tmp_ts_s == ts_s:
            #     log(f"Duplicate streamer frames")
            fetch_s = False

        tmp_ts_p = 0
        if fetch_p:
            frame_p = get_frame(pipe_p)
            if frame_p is None:
                log(f"End video player")
                break
            tmp_ts_p = parse_timestamp(frame_p, 7)
            if tmp_ts_p < args.start:
                log("Skip forward in video 2 ...")
                continue
            if tmp_ts_p < 0 or tmp_ts_p < ts_p: # could not read the ts / parsed earlier than current ts_p; must be a parsing error
                ts_p += 1
            elif tmp_ts_p == ts_p: # either frame is stuck (on player does not really happen) or sender
                pass
            else:
                ts_p = tmp_ts_p
                fetch_p = False

        if args.stop > 0 and ts_s >= args.stop:
            log(f"Stopping due to --stop={args.stop}")
            break

        ssim = 0
        msg = None
        ssim = compute_ssim(frame_s, frame_p)
        if fetch_p and not fetch_s:
            if tmp_ts_p == 0:
                msg = f"failed to parse ts_p ({tmp_ts_p})"
            elif prev_ts_s == ts_s:
                msg = "duplicate streamer frame"
            elif tmp_ts_p == ts_p:
                msg = "duplicate player frame"
            else:
                msg = f"parsed earlier than current ts_p ({tmp_ts_p})"
            fetch_s = True
        elif ts_s == ts_p:
            start_up = False
            fetch_s = True
            fetch_p = True
        elif ts_s > ts_p:
            ssim = 0
            msg = "no player frame"
            fetch_p = True
        elif ts_s < ts_p and start_up:
            ssim = 0
            msg = "Catching up to player"
            fetch_s = True
        elif ts_s < ts_p:
            msg = "catching up to player"
            fetch_s = True
        else:
            breakpoint()

        writer(ts_s, ts_p, ssim, msg)
        if msg is not None:
            log(f"ssim={ssim:.6f} ({msg})")
        elif ssim == 1:
            ret = '\r' if need_newline else ''
            sys.stdout.write(f"{ret}{get_prefix()} ssim=1.0")
            need_newline = True
        else:
            log(f"ssim={ssim}")

    writer(None, None, None, None)


if __name__ == '__main__':
    main()
