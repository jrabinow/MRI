#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>

#ifndef UTILS_H
#define UTILS_H

void *xmalloc(size_t size);

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
void failwith(char *errmsg);

#endif
