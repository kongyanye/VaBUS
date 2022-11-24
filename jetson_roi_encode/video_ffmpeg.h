#ifndef FFMPEG_H
#define FFMPEG_H

#include "NvBuffer.h"

#include <fstream>
#include <iostream>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

int ffmpeg_read_video_frame(std::ifstream * stream, NvBuffer & buffer);
void ffmpeg_init( uint32_t srcW, uint32_t srcH, AVPixelFormat srcPixFmt );


#endif


