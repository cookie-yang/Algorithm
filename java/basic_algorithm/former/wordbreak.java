public class wordbreak{
    public boolean wordBreak(String s, Set<String> dict) {
        boolean[] flaglist = new boolean[s.length() + 1];
        for (boolean tmp : flaglist) {
            tmp = false;
        }
        flaglist[0] = true;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) {
                if (flaglist[i] == true && dict.contains(s.substring(i, j))) {
                    flaglist[j] = true;
                }
            }
        }
        if (flaglist[s.length()] == true)
            return true;
        else
            return false;

    }
    public static void main(String[] args) {
        
    }
};