public class full_permutation_recursion{
    public static void swap(char[] charlist, int start, int end) {
        char tmp = charlist[start];
        charlist[start] = charlist[end];
        charlist[end] = tmp;
    }

    public static void full_permutation(char[] charlist, int start) {
        if (start == charlist.length) {
            for (int i = 0; i < charlist.length; i++)
                System.out.print(charlist[i]);
            System.out.println();
        } else {
            for (int i = start; i < charlist.length; i++) {
                boolean exist = false;
                for (int j = start; j < i; j++)
                    if (charlist[j] == charlist[i])
                        exist = true;
                if (exist == false) {
                    swap(charlist, start, i);
                    full_permutation(charlist, start + 1);
                    swap(charlist, start, i);
                }

            }
        }
    }
    public static void main(String[] args) {
        char[] charList = { 'a', 'a', 'c', 'd' };
        full_permutation(charList, 0);
    }
}