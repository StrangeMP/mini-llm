"""
Note: a correct and efficient implementation of BPE is non-trivial. 
The following is a simplified pseudocode of BPE for reference. 
1. pre-tokenize: run regex on the corpus, resulting pretoks = ['aba', 'aab', ...] 
2. correspondingly let groups = [['a', 'b', 'a'], ...] 
3. count adjacent pairs : [('a', 'b', 2), ('b', 'a', 1), ('a', 'a', 1), ...] 
4. retrieve most frequent pairs: (x, y, count) 
5. add x + y to vocabulary 
6. merge x and y in groups: (p, x, y, s) -> (p, xy, s) 
7. repeat 3 to 6 until the stopping criteria is met
"""