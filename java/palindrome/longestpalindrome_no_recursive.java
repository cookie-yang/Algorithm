import java.util.*;
public class longestpalindrome_no_recursive{
    public static int longestpalindrome(String s, int i, int j ,Integer[][] array){
        if(array[i][j]!=false)
        return array[i][j];
        if(i==j)return 1;
        if(i>j)return 0;
        if(s.charAt(i)==s.charAt(j))
        array[i][j] = longestpalindrome(s,i+1,j-1,array)+2;
        else
        array[i][j]=Math.max(longestpalindrome(s,i+1,j,array),longestpalindrome(s,i,j-1,array));
        
        return array[i][j];
    }
    public static void main(String[] args){
        String s = "abba";
        longestpalindrom(s,0,s.length()-1,new Integer[s.length()][s.length()]);
    }
}