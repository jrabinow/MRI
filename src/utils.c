#include <utils.h>

void *xmalloc(size_t size)
{
	void *ptr = NULL;

	ptr = malloc(size);
	if(ptr != NULL)
		return ptr;
	perror("Error allocating memory ");
	exit(EXIT_FAILURE);
}

void *xrealloc(void *ptr, size_t size)
{
	ptr = realloc(ptr, size);
	if(ptr != NULL)
		return ptr;
	perror("Error allocating memory ");
	exit(EXIT_FAILURE);
}

FILE *xfopen(const char *path, const char *mode)
{
	FILE *f = (FILE*) NULL;

	f = fopen(path, mode);
	if(f != NULL)
		return f;
	perror("Error opening file ");
	exit(EXIT_FAILURE);
}


void failwith(const char *errmsg)
{
	fputs(errmsg, stderr);
	fputc('\n', stderr);
	exit(EXIT_FAILURE);
}

static log_level_t __g_loglevel = LOG_DEBUG;
static FILE *__g_loghandle = NULL;

void init_log(FILE *stream, log_level_t loglevel)
{
	__g_loglevel = loglevel;
	__g_loghandle = stream;
}

void log_message(log_level_t level, const char *format, ...)
{
	char buffer[255] = { 0 }, timestamp[255] = { 0 };
	const char *slevel;
	va_list ap;
	time_t rawtime;
	struct tm *timeinfo;

	if(__g_loghandle != NULL && level >= __g_loglevel) {
		va_start(ap, format);
		vsnprintf(buffer, 255, format, ap);
		va_end(ap);
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(timestamp, 255, "%d/%m/%Y %X", timeinfo);

		switch(level) {
			case LOG_DEBUG:
				slevel = "DEBUG";
				break;
			case LOG_INFO:
				slevel = "INFO";
				break;
			case LOG_WARNING:
				slevel = "WARNING";
				break;
			case LOG_ERROR:
				slevel = "ERROR";
				break;
			case LOG_FATAL:
			default:
				slevel = "FATAL";
				break;
		}
		fprintf(__g_loghandle, "[%s] [%s] %s\n", timestamp, slevel, buffer);
	}
}

