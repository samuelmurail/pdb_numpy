#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MATRIX_SIZE 20
#define GAP_COST -11
#define GAP_EXT -1

#define TRUE 1
#define FALSE 0

typedef struct {
    char *seq1;
    char *seq2;
    int score;
} Alignment;


int* seq_to_num (char *seq)
{
    int seq_len = strlen(seq);
    int *seq_num = malloc(seq_len * sizeof(int));
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

void print_alignment(Alignment alignment)
{
    int align_len = strlen(alignment.seq1);
    int line_len = 80;
    printf("Alignment score: %d\n", alignment.score);

    for (int line = 0; line < (align_len / line_len + 1); line++)
    {
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%c", alignment.seq1[i]);
        }
        printf("\n");
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%s",  (alignment.seq1[i] == alignment.seq2[i] ? "*" : " "));
        }
        printf("\n");
        for (int i = line * line_len; (i < (line + 1) * line_len) && (i < align_len); i++)
        {
            printf("%c", alignment.seq2[i]);
        }
        printf("\n");
        //if (alignment.seq1[i] == alignment.seq2[i]) printf("|");
        //else printf(" ");
    }


    return ;
}

Alignment align(char *seq1, char *seq2, short int subs_matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    int seq1_len = strlen(seq1), seq2_len = strlen(seq2);
    int score_matrix [seq1_len + 1][seq2_len + 1];
    int prev_score_line [seq2_len];
    int prev_score = FALSE;
    int i, j, counter;

    Alignment alignment;

    for (i = 0; i < seq1_len + 1; i++) score_matrix[i][0] = 0;
    for (i = 0; i < seq2_len + 1; i++) score_matrix[0][i] = 0;
    for (i = 0; i < seq2_len; i++) prev_score_line[i] = FALSE;


    int *seq_1_num = seq_to_num(seq1);
    int *seq_2_num = seq_to_num(seq2);

    int match, insert, delete;

    printf("Aligning: \n%s \nwith:\n%s\n", seq1, seq2);

    for (int i = 1; i < seq1_len + 1; i++)
    {
        prev_score = FALSE;
        for (int j = 1; j < seq2_len + 1; j++)
        {
            match = score_matrix[i-1][j-1] + subs_matrix[seq_1_num[i-1]][seq_2_num[j-1]];
            insert = score_matrix[i][j-1] + (prev_score_line[j] ? GAP_EXT : GAP_COST);
            delete = score_matrix[i -1][j] + (prev_score ? GAP_EXT : GAP_COST);

            if (match > insert && match > delete) {
                score_matrix[i][j] = match;
                prev_score_line[j] = FALSE;
                prev_score = FALSE;
            } else if (insert > delete) {
                score_matrix[i][j] = insert;
                prev_score_line[j] = FALSE;
                prev_score = TRUE;
            } else {
                score_matrix[i][j] = delete;
                prev_score_line[j] = TRUE;
                prev_score = FALSE;
            }
        }
    }

    // Get max score index:
    int min_len = (seq1_len < seq2_len) ? seq1_len : seq2_len;
    int min_i=0, min_j=0, max_score = score_matrix[0][0];

    for (int i = min_len; i < seq1_len + 1; i++)
    {
        for (int j = min_len; j < seq2_len + 1; j++)
        {
            if (score_matrix[i][j] > max_score) {
                max_score = score_matrix[i][j];
                min_i = i;
                min_j = j;
            }
        }
    }

    alignment.score = max_score;

    // Traceback and compute the alignment

    int max_len = seq1_len + seq2_len;
    char *align_seq_1 = malloc(max_len * sizeof(char));
    char *align_seq_2 = malloc(max_len * sizeof(char));

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

    do {
        if (score_matrix[i+1][j+1] == score_matrix[i][j] + subs_matrix[seq_1_num[i]][seq_2_num[j]]) {
            align_seq_1[counter] = seq1[i--];
            align_seq_2[counter] = seq2[j--];
        }
        else if ((score_matrix[i+1][j+1] == score_matrix[i][j+1] + GAP_COST) ||
                 (score_matrix[i+1][j+1] == score_matrix[i][j+1] + GAP_EXT)) {
            align_seq_1[counter] = seq1[j--];
            align_seq_2[counter] = '-';
        }
        else {
            align_seq_2[counter] = seq2[i--];
            align_seq_1[counter] = '-';
        }
        counter ++;
    } while (i >= 0 && j >= 0);

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

    // Inverse the sequences
    alignment.seq1 = malloc(max_len * sizeof(char));
    alignment.seq2 = malloc(max_len * sizeof(char));

    int rev_i = 0;
    for (int i = counter; i >= 0; i--) {
        if (align_seq_1[i] == '\0')
            continue;
        alignment.seq1[rev_i] = align_seq_1[i];    
        alignment.seq2[rev_i] = align_seq_2[i];
        rev_i++;
    }
    alignment.seq1[rev_i] = '\0';
    alignment.seq2[rev_i] = '\0';

    free(align_seq_1);
    free(align_seq_2);

    return alignment;
}

void read_matrix(char *matrix_file, short int matrix[MATRIX_SIZE][MATRIX_SIZE])
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

    short int sub_matrix[MATRIX_SIZE][MATRIX_SIZE];
    printf("Hello World!\n");
    
    
    read_matrix("data/blosum62.txt", sub_matrix);

    printf ("MATRIX:\n");
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            printf("%2d ", sub_matrix[i][j]);
        }
        printf("\n");
    }

    Alignment alignment;
    alignment = align(seq_1, seq_2, sub_matrix);
    printf ("Alignment:\n%s:\n%s:\n", alignment.seq1, alignment.seq2);

    print_alignment(alignment);

    return EXIT_SUCCESS;

}