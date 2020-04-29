/* LOAD_ALIST.H - Loads ALIST file. */
//Module created by Joschka Roffe. Based on code by Radford M. Neal (See copyright notice below)

/* Copyright (c) 1995-2012 by Radford M. Neal.
 *
 * Permission is granted for anyone to copy, use, modify, and distribute
 * these programs and accompanying documents for any purpose, provided
 * this copyright notice is retained and prominently displayed, and note
 * is made of any changes made to these programs.  These programs and
 * documents are distributed without any warranty, express or implied.
 * As the programs were written for research purposes only, they have not
 * been tested to the degree that would be advisable in any important
 * application.  All use of these programs is entirely at the user's own
 * risk.
 */

void bad_alist_file(void);
void usage(void);

mod2sparse *load_alist (char *alist_file);
