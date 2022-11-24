#include "video_ffmpeg.h"

struct SwsContext *sws_ctx;
uint8_t *src_data[4];
uint8_t *dst_data[4];
int src_linesize[4];
int dst_linesize[4];
uint32_t src_w = 720;
uint32_t src_h = 406;
uint32_t dst_w = 720;
uint32_t dst_h = 406;
enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_BGR24;
enum AVPixelFormat dst_pix_fmt = AV_PIX_FMT_YUV420P;

int
ffmpeg_read_video_frame(std::ifstream * stream, NvBuffer & buffer)
{
    uint32_t i, j;
	std::streamsize read_bytes;
	char *plane_data;

	read_bytes = 3*src_w*src_h;
    char *data = new char [read_bytes];
    
	stream->read(data, read_bytes);
    if (stream->gcount() < read_bytes)
  	{
		std::cout << "read_bytes is " << read_bytes << ", but stream gcount is " << stream->gcount() << std::endl;
		return -1;
	}
	
	memcpy(src_data[0], data, read_bytes);
	sws_scale(sws_ctx, (const uint8_t* const*)src_data, src_linesize, 0, src_h, dst_data, dst_linesize);
	
	for (i = 0; i < buffer.n_planes; i++)
    {
        NvBuffer::NvBufferPlane &plane = buffer.planes[i];
        std::streamsize bytes_to_read =
            plane.fmt.bytesperpixel * plane.fmt.width;
        plane_data = (char *) plane.data;
        plane.bytesused = 0;
        for (j = 0; j < plane.fmt.height; j++)
        {
            memcpy(plane_data, &(dst_data[i][j*bytes_to_read]), bytes_to_read);
		    plane_data += plane.fmt.stride;
        }
        plane.bytesused = plane.fmt.stride * plane.fmt.height;
		//std::cout << "heigth is " << plane.fmt.height << ", stride is " << plane.fmt.stride << ", bytesperpixel * width = " << plane.fmt.bytesperpixel << " * " << plane.fmt.width << " = " << bytes_to_read << ", bufferplane is " << i << std::endl;
    }

	delete[] data;
	
    return 0;
}

void ffmpeg_init( uint32_t srcW, uint32_t srcH, AVPixelFormat srcPixFmt )
{
	src_w = srcW;
	src_h = srcH;
	dst_w = src_w;
	dst_h = src_h;
	src_pix_fmt = srcPixFmt;

	//std::cout << "srcW is " << src_w << " srcH is " << src_h << " dst_w is " << dst_w << " dst_h is " << dst_h << " pixfmt is " << src_pix_fmt << std::endl; 

	avdevice_register_all();
	sws_ctx = sws_getContext(src_w, src_h, src_pix_fmt, dst_w, dst_h, dst_pix_fmt, SWS_BICUBIC, NULL, NULL, NULL);
	if ( !sws_ctx )
	{
		std::cout << "impossible to create scale context for conversion" << std::endl;
		sws_freeContext(sws_ctx);
	}

	if( av_image_alloc(src_data, src_linesize, src_w, src_h, src_pix_fmt, 1) < 0 )
	{
		std::cout << "Could not allocate source image" << std::endl;
	}

	if( av_image_alloc(dst_data, dst_linesize, dst_w, dst_h, dst_pix_fmt, 1) < 0 )
	{
		std::cout << "Could not allocate destination image" << std::endl;
	}
}

