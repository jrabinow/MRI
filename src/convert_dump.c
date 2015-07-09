#include <utils.h>
#include <cuComplex.h>

typedef struct {
	cuDoubleComplex val;
	char valid;
} LineData;

LineData parse_line(FILE *stream)
{
	LineData ld;
	float r, i;
	size_t linecpt = 1;

	ld.valid = 0;
	if(fscanf(stream, "%f%*3c%f", &r, &i) != 2) {
		switch(ferror(stream)) {
			case 0:
				break;
			default:
				perror("Error parsing line");
				printf("Line number: %zu\n", linecpt);
				fcloseall();
				exit(EXIT_FAILURE);
		}
		linecpt++;
	} else
		ld.valid = 1;
	empty_buffer(stream);
	ld.val = make_cuDoubleComplex(r, i);

	return ld;
}

int main(int argc, char **argv)
{
	FILE *in = NULL, *out = NULL;
	LineData ld;

	if(argc != 3)
		failwith("Usage: convert_dump INPUT OUTPUT");
	in = xfopen(argv[1], "r");
	out = xfopen(argv[2], "w");

	while((ld = parse_line(in)).valid)
		fwrite(&ld.val, sizeof(ld.val), 1, out);
	fclose(in);
	fclose(out);
	return 0;
}
