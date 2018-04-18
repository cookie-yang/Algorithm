import util.java.ArrayList;
public class Substring_with_Concatenation_of_All_Words {
    public static List<Integer> findSubstring(String s, String[] words) {
        int wordLength = words[0].length();
        int wordNum = words.length;
        int[] wordIndex = new int[wordNum];
        List<Integer> result = new ArrayList<Integer>();
        int i = 0;
        for (String word : words) {
            wordIndex[i] = s.indexOf(word);
            i++;
        }
        for(i=0;i<wordNum;i++){
            String tmpStr = s.substring(wordIndex[i], wordIndex[i]+wordLength*wordNum);
            int j = 0;
            for(;j<wordNum;j++){
                if(tmpStr.contains(words[j])==false)
                break;
            }
            if(j==(wordNum-1)){
                // System.out.print(j);
                result.add(wordIndex[i]);
            }
        }
    return result;
    }
    public static void main(String[] args) {
        String s = "barfoothefoobarman";
        String [] words = {"foo","bar"};
        findSubstring(s,words);
}
}