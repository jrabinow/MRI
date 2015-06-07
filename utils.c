#include "utils.h"

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


void failwith(char *errmsg)
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
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
	Color color = WHITE, bgcolor = BLACK;
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */

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
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
				color = GREEN;
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */
				break;
			case LOG_WARNING:
				slevel = "WARNING";
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
				color = YELLOW;
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */
				break;
			case LOG_ERROR:
				slevel = "ERROR";
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
				color = RED;
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */
				break;
			case LOG_FATAL:
			default:
				slevel = "FATAL";
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
				bgcolor = RED;
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */
				break;
		}
#if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix)
		if(__g_loghandle == stdout || __g_loghandle == stderr) {
			fprintf(__g_loghandle, "\x1B[%d;0m", color + 30);
			fprintf(__g_loghandle, "\x1B[%dm[%s] [%s] %s", bgcolor + 40, timestamp, slevel, buffer);
			reset_style(__g_loghandle);
			putc('\n', __g_loghandle);
			fflush(__g_loghandle);
		} else
#endif	/* #if defined(ENABLE_TERMIOS_MANIPULATION) && defined(__unix) */
			fprintf(__g_loghandle, "[%s] [%s] %s\n", timestamp, slevel, buffer);
	}
}

