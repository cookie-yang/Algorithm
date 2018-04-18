public class workbreak2{
    public static List<String> wordBreak(String s, List<String> wordDict) {
        List<String> res = new ArrayList<String>();
        HashMap<String, List<String>> sl = new HashMap<String, List<String>>();
        getfirstword(s, res, wordDict, sl);
        return sl.get(s);
    }

    public static void getfirstword(String s, List<String> res, List<String> wordDict,
            HashMap<String, List<String>> sl) {
        if (sl.containsKey(s)) {
            for (String tmpstr : sl.get(s))
                res.add(tmpstr);
        }

        else if (s.length() == 0) {
            res.add("");

        } else {

            for (String str : wordDict) {
                if (s.startsWith(str)) {
                    List<String> newLink = new ArrayList<String>();
                    getfirstword(s.substring(str.length()), newLink, wordDict, sl);
                    for (String str1 : newLink) {
                        String newstr = str + " " + str1;
                        res.add(newstr.trim());
                    }
                }
            }
            sl.put(s, res);
        }

    }
    public static void main(String[] args) {
        String s = "catsanddog";
        String[] str = new String[] { "a", "cat", "cats", "and", "sand", "dog" };
        List<String> wordDict = new ArrayList<String>();
        wordDict.add("cat");
        wordDict.add("cats");
        wordDict.add("sand");
        wordDict.add("and");
        wordDict.add("dog");
        List<String> wordDict1 = wordBreak(s, wordDict);
        for (String str1 : wordDict1) {
            System.out.println(str1);
        }
    }
}