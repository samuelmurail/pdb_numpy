#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MATRIX_SIZE 20

#define TRUE 1
#define FALSE 0

typedef struct {
    char *seq1;
    char *seq2;
    int score;
} Alignment;


int* seq_to_num (const char *seq)
{
    int seq_len = strlen(seq);
    int *seq_num = malloc(seq_len * sizeof(int));
    assert(seq_num != NULL);
    for (int i = 0; i < seq_len; i++)
    {
        switch (seq[i])
        {
            case 'A':
                seq_num[i] = 0;
                break;
            case 'R':
                seq_num[i] = 1;
                break;
            case 'N':
                seq_num[i] = 2;
                break;
            case 'D':
                seq_num[i] = 3;
                break;
            case 'C':
                seq_num[i] = 4;
                break;
            case 'Q':
                seq_num[i] = 5;
                break;
            case 'E':
                seq_num[i] = 6;
                break;
            case 'G':
                seq_num[i] = 7;
                break;
            case 'H':
                seq_num[i] = 8;
                break;
            case 'I':
                seq_num[i] = 9;
                break;
            case 'L':
                seq_num[i] = 10;
                break;
            case 'K':
                seq_num[i] = 11;
                break;
            case 'M':
                seq_num[i] = 12;
                break;
            case 'F':
                seq_num[i] = 13;
                break;
            case 'P':
                seq_num[i] = 14;
                break;
            case 'S':
                seq_num[i] = 15;
                break;
            case 'T':
                seq_num[i] = 16;
                break;
            case 'W':
                seq_num[i] = 17;
                break;
            case 'Y':
                seq_num[i] = 18;
                break;
            case 'V':
                seq_num[i] = 19;
                break;
            default:
                seq_num[i] = -1;
                break;
        }
    }

    return seq_num;
}

void print_alignment(Alignment *alignment)
{
    int align_len = strlen(alignment->seq1);
    int align_len2 = strlen(alignment->seq2);
    if (align_len != align_len2)
    {
        printf("Error: alignment length mismatch (%d, %d) !\n", align_len, align_len2);
        return;
    }
    int line_len = 80;
    printf("Alignment score: %d\n", alignment->score);

    for (int line = 0; line < (align_len / line_len) + 1; line++)
    {
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%c", alignment->seq1[i]);
        }
        printf("\n");
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%c",  (alignment->seq1[i] == alignment->seq2[i] ? '*' : ' '));
        }
        printf("\n");
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%c", alignment->seq2[i]);
        }
        printf("\n");

    }
    return ;
}

void read_matrix(const char *matrix_file, short int matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    //short int matrix[MATRIX_SIZE][MATRIX_SIZE];
    char *ptr;
    char delim[] = " ";

    fp = fopen(matrix_file, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int i = 0, j;
    while (((read = getline(&line, &len, fp)) != -1) && (i < MATRIX_SIZE)) {
        if (line[0] == '#' || line[0] == ' ' || line[0] == '*')
            continue;
        if (i >= 0) {
            ptr = strtok(line, delim);
            ptr = strtok(NULL, delim);
            for (j = 0; (ptr != NULL) && (j < MATRIX_SIZE); j++)
            {
                matrix[i][j] = atoi(ptr);
                ptr = strtok(NULL, delim);
            }
        }
        i++;
    }

    // Free dynamically allocated memory for line
    free(line);

    // Close file
    fclose(fp);
}

Alignment *align_test(const char *seq1, const char *seq2, const char *matrix_file, int GAP_COST, int GAP_EXT)
{
    Alignment *alignment = malloc(sizeof(Alignment));
    assert(alignment != NULL);
    alignment->seq1 = malloc((strlen(seq1) + 1) * sizeof(char));
    assert(alignment->seq1 != NULL);
    alignment->seq2 = malloc((strlen(seq2) + 1) * sizeof(char));
    assert(alignment->seq2 != NULL);
    strcpy(alignment->seq1, seq1);
    strcpy(alignment->seq2, seq2);
    alignment->score = 0;
    return alignment;
}

void check_seq(const char *seq)
{
    int len = strlen(seq);
    for (int i = 0; i < len; i++)
    {
        if ((seq[i] < 'A' || seq[i] > 'Z') && seq[i] != '-')
        {
            printf("Invalid character '%c'|%d pos=%d\n", seq[i], seq[i], i);
            printf("Full sequence =  %s\n", seq);
            exit(EXIT_FAILURE);
        }
    }
}

//void align(const char *seq1, const char *seq2, const char *matrix_file, int GAP_COST, int GAP_EXT)


Alignment *align(const char *seq1, const char *seq2, const char *matrix_file, int GAP_COST, int GAP_EXT)
{
    short int subs_matrix[MATRIX_SIZE][MATRIX_SIZE];
    int seq1_len = strlen(seq1), seq2_len = strlen(seq2);
    int score_matrix [seq1_len + 1][seq2_len + 1];
    int prev_score_line [seq2_len];
    int prev_score = FALSE;
    int i, j, counter;
    Alignment *alignment = malloc(sizeof(Alignment));
    assert(alignment != NULL);

    // clean input sequences
    check_seq(seq1);
    check_seq(seq2);

    read_matrix(matrix_file, subs_matrix);

    for (i = 0; i < seq1_len + 1; i++) score_matrix[i][0] = 0;
    for (i = 0; i < seq2_len + 1; i++) score_matrix[0][i] = 0;
    for (i = 0; i < seq2_len; i++) prev_score_line[i] = FALSE;


    int *seq_1_num = seq_to_num(seq1);
    int *seq_2_num = seq_to_num(seq2);

    int match, insert, delete;


    for (int i = 1; i <= seq1_len; i++)
    {
        prev_score = FALSE;
        for (int j = 1; j <= seq2_len; j++)
        {
            match = score_matrix[i - 1][j - 1] + subs_matrix[seq_1_num[i-1]][seq_2_num[j-1]];
            delete = score_matrix[i - 1][j] + (prev_score ? GAP_EXT : GAP_COST);
            insert = score_matrix[i][j - 1] + (prev_score_line[j] ? GAP_EXT : GAP_COST);

            if (match > insert && match > delete) {
                score_matrix[i][j] = match;
                prev_score_line[j] = FALSE;
                prev_score = FALSE;
            } else if (delete > insert) {
                score_matrix[i][j] = delete;
                prev_score_line[j] = FALSE;
                prev_score = TRUE;
            } else {
                score_matrix[i][j] = insert;
                prev_score_line[j] = TRUE;
                prev_score = FALSE;
            }
        }
    }

    // Get max score index:
    int min_len = (seq1_len < seq2_len) ? seq1_len : seq2_len;
    int min_i=0, min_j=0, max_score = score_matrix[0][0];

    //int show_num = 15;
    //for (int i = 0; i < show_num; i++)
    //{
    //    for (int j = 0; j < show_num; j++)
    //    {
    //        printf ("%3d ", score_matrix[i][j]);
    //    }
    //    printf("\n");
    //}

    //int show_num = 10;
    //for (int i = seq1_len-show_num; i <= seq1_len; i++)
    //{
    //    for (int j = seq2_len-show_num; j <= seq2_len; j++)
    //    {const char *seq1, const char *seq2
    //        printf ("%3d ", score_matrix[i][j]);
    //    }
    //    printf("\n");
    //}
    //printf ("Get max \n");
    for (int i = min_len; i <= seq1_len; i++)
    {
        for (int j = min_len; j <= seq2_len; j++)
        {
            //printf ("%3d ", score_matrix[i][j]);
            if (score_matrix[i][j] > max_score) {
                max_score = score_matrix[i][j];
                min_i = i;
                min_j = j;
            }
        }
        //printf("\n");
    }

    //printf ("Max score: %d at %d, %d\n", max_score, min_i, min_j);

    alignment->score = max_score;

    // Traceback and compute the alignment

    int max_len = seq1_len + seq2_len;
    char align_seq_1[max_len];
    char align_seq_2[max_len];

    //printf ("A Checking align sequences 1 len = %ld:\n", strlen(align_seq_1));
    //check_seq(align_seq_1);
    //printf ("A Checking align sequences 2 len = %ld:\n", strlen(align_seq_2));
    //check_seq(align_seq_2);

    counter = 0;
    i = min_i;
    j = min_j;

    for (i = seq1_len - 1; i >= min_i; i--) {
        align_seq_1[counter] = seq1[i];
        align_seq_2[counter] = '-';
        counter++;
    }

    for (j = seq2_len - 1; j >= min_j; j--) {
        align_seq_2[counter] = seq2[j];
        align_seq_1[counter] = '-';
        counter++;
    }

    align_seq_1[counter] = '\0';
    align_seq_2[counter] = '\0';

    //printf ("B Checking align sequences 1 len = %ld:\n", strlen(align_seq_1));
    //check_seq(align_seq_1);
    //printf ("B Checking align sequences 2 len = %ld:\n", strlen(align_seq_2));
    //check_seq(align_seq_2);

    //printf ("Start matrix backtrack i=%d, j=%d, counter=%d\n", i, j, counter);
    do {
        if (score_matrix[i+1][j+1] == score_matrix[i][j] + subs_matrix[seq_1_num[i]][seq_2_num[j]]) {
            align_seq_1[counter] = seq1[i--];
            align_seq_2[counter] = seq2[j--];
            //printf ("Match: %c, %c at i=%d, j=%d counter=%d\n", align_seq_1[counter], align_seq_2[counter], i, j, counter);
        }
        else if ((score_matrix[i+1][j+1] == score_matrix[i][j+1] + GAP_COST) ||
                 (score_matrix[i+1][j+1] == score_matrix[i][j+1] + GAP_EXT)) {
            align_seq_1[counter] = seq1[i--];
            align_seq_2[counter] = '-';
            //printf ("Insert: %c, %c at i=%d, j=%d counter=%d\n", align_seq_1[counter], align_seq_2[counter], i, j, counter);
        }
        else {
            align_seq_1[counter] = '-';
            align_seq_2[counter] = seq2[j--];
            //printf ("Delete: %c, %c at i=%d, j=%d counter=%d\n", align_seq_1[counter], align_seq_2[counter], i, j, counter);
        }
        counter ++;
    } while (i >= 0 && j >= 0);

    align_seq_1[counter] = '\0';
    align_seq_2[counter] = '\0';

    //printf ("C Checking align sequences 1 len = %ld:\n", strlen(align_seq_1));
    //check_seq(align_seq_1);
    //printf ("C Checking align sequences 2 len = %ld:\n", strlen(align_seq_2));
    //check_seq(align_seq_2);

    for (i = i + 1; i > 0; i--) {
        align_seq_1[counter] = seq1[i - 1];
        align_seq_2[counter] = '-';
        counter ++;
    }
    for (j = j + 1; j > 0; j--) {
        align_seq_2[counter] = seq2[j - 1];
        align_seq_1[counter] = '-';
        counter ++;
    }
    align_seq_1[counter] = '\0';
    align_seq_2[counter] = '\0';


    //printf ("D Checking align sequences 1 len = %ld:\n", strlen(align_seq_1));
    //check_seq(align_seq_1);
    //printf ("D Checking align sequences 2 len = %ld:\n", strlen(align_seq_2));
    //check_seq(align_seq_2);
    // Get size of the alignment
    int align_len = 0;
    for (int i = 0; i < max_len; i++) {
        if ((align_seq_1[i] == '\0') || (align_seq_2[i] == '\0')) {
            break;
        }
        align_len++;
    }

    // Inverse the sequences
    alignment->seq1 = calloc((align_len + 1), sizeof(char));
    alignment->seq2 = calloc((align_len + 1), sizeof(char));
    assert(alignment->seq1 != NULL);
    assert(alignment->seq2 != NULL);
    //printf ("Alignment len align_len: %d\n", align_len + 1);

    int rev_i = 0;
    for (int i = counter - 1; i >= 0; i--) {
        if ((align_seq_1[i] == '\0') || (align_seq_2[i] == '\0')) {
            continue;
        }
        alignment->seq1[rev_i] = align_seq_1[i];    
        alignment->seq2[rev_i] = align_seq_2[i];
        //printf ("%c %c %d %d %d\n", alignment->seq1[rev_i], alignment->seq2[rev_i], alignment->seq1[rev_i], alignment->seq2[rev_i], rev_i);
        rev_i++;
    }
    alignment->seq1[rev_i] = '\0';
    alignment->seq2[rev_i] = '\0';

    //printf ("Alignment len %d\n", rev_i);

    free(seq_1_num);
    free(seq_2_num);
    check_seq(alignment->seq1);
    check_seq(alignment->seq2);

    //print_alignment(alignment);

    return alignment;
}

void free_align(Alignment *align)
{
    if (align == NULL) {
        printf("Warning: alignment is NULL in free_align function !\n");
        return;
    }

    //printf("Freeing alignment seq1: %s\n", align->seq1);
    if (align->seq1 != NULL) {
        free(align->seq1); 
        align->seq1 = NULL;
    } else {
        printf("Warning: seq1 is NULL\n");
    }
    //printf("Freeing alignment seq2: %s\n", align->seq2);
    if (align->seq2 != NULL) {
        free(align->seq2);
        align->seq2 = NULL;
    } else {
        printf("Warning: seq2 is NULL\n");
    }

    // Free memory for the alignment structure
    free(align);

}

void test(char *seq_1, char *seq_2) {

    printf("Test\n");

    Alignment *alignment = NULL;
    
    alignment = align(seq_1, seq_2, "../data/blosum62.txt", -11, -1);
    printf ("Alignment:\n%s:\n%s:\n", alignment->seq1, alignment->seq2);

    //print_alignment(alignment);

    free_align(alignment);


}

int main(int argc, char *argv[])
{
    char *seq_1 = "AQDMVSPPPPIADEPLTVNTGIYLIECYSLDDKAETFKVNAFLSLSWKDRRLAFDPV\
RSGVRVKTYEPEAIWIPEIRFVNVENARDADVVDISVSPDGTVQYLERFSARVLSPLDFRRYPFDSQTLHIYLIVR\
SVDTRNIVLAVDLEKVGKNDDVFLTGWDIESFTAVVKPANFALEDRLESKLDYQLRISRQYFSYIPNIILPMLFIL\
FISWTAFWSTSYEANVTLVVSTLIAHIAFNILVETNLPKTPYMTYTGAIIFMIYLFYFVAVIEVTVQHYLKVESQP\
ARAASITRASRIAFPVVFLLANIILAFLFFGF";
    char *seq_2 = "MFALGIYLWETIVFFSLAASQQAAARKAASPMPPSEFLDKLMGKVSGYDARIRPNFK\
GPPVNVTCNIFINSFGSIAETTMDYRVNIFLRQQWNDPRLAYSEYPDDSLDLDPSMLDSIWKPDLFFANEKGANFH\
EVTTDNKLLRISKNGNVLYSIRITLVLACPMDLKNFPMDVQTCIMQLESFGYTMNDLIFEWDEKGAVQVADGLTLP\
QFILKEEKDLRYCTKHYNTGKFTCIEARFHLERQMGYYLIQMYIPSLLIVILSWVSFWINMDAAPARVGLGITTVL\
TMTTQSSGSRASLPKVSYVKAIDIWMAVCLLFVFSALLEYAAVNFIARQHKELLRFQRRRRHLKEDEAGDGRFSFA\
AYGMGPACLQAKDGMAIKGNNNNAPTSTNPPEKTVEEMRKLFISRAKRIDTVSRVAFPLVFLIFNIFYWITYKIIR\
SEDIHKQ";
    /*char *seq_1 = "AQDMVSPPPPIADEPLTVNT";
    char *seq_2 = "VSPPPPIADEP";*/

    //short int sub_matrix[MATRIX_SIZE][MATRIX_SIZE];
    printf("Hello World!\n");
    
    printf ("READ MATRIX:\n");

    /*read_matrix("../data/blosum62.txt", sub_matrix);

    printf ("MATRIX:\n");
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            printf("%2d ", sub_matrix[i][j]);
        }
        printf("\n");
    }*/

    //Alignment *alignment;

    //free(alignment->seq2);
    //free(alignment->seq1);

    int iter_num = 1000;

    for (int i = 0; i < iter_num; i++) {
        printf("Test %d\n", i);
        test(seq_1, seq_2);
    }


    return EXIT_SUCCESS;

}