#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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

char *const_append(const char *str1, const char *str2);
char *itoa(int n, char *buffer);

/* Empties buffer till nothing left to read or hits end of line. Useful with scanf/fscanf */
#define empty_buffer(stream)    {\
	int __c__;\
	while((__c__ = getc(stream)) != EOF && __c__ != '\n');\
}

#ifdef __cplusplus
}
#endif
#endif
