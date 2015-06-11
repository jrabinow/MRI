#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C"
{
#endif

void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t size);
FILE *xfopen(const char *path, const char *mode);

typedef enum {
	LOG_DEBUG,
	LOG_INFO,
	LOG_WARNING,
	LOG_ERROR,
	LOG_FATAL
} log_level_t;

/* Initialize log system on file stream */
void init_log(FILE *stream, log_level_t loglevel);
/* logs a message following printf conventions */
void log_message(log_level_t level, const char *format, ... );

/* prints errmsg to stderr and calls exit(). Functions previously
 * registered with atexit() will be called */
void failwith(const char *errmsg);

#ifdef __cplusplus
}
#endif
#endif
