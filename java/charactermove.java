import java.util.Scanner;

public class charactermove {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        StringBuilder n = new StringBuilder(in.nextLine());
        int end = n.length();
        for (int i = 0; i < end; i++) {
            if (Integer.valueOf(n.charAt(i)) < 97) {
                char tmp = n.charAt(i);
                n.deleteCharAt(i);
                n.append(tmp);
                i--;
                end--;
            }
        }
        System.out.print(n.toString());
    }
}