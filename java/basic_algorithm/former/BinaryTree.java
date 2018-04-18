public class BinaryTree{
   	public static class TreeNode{
	    public StringBuffer value = new StringBuffer();
	    public TreeNode leftson;
	    public TreeNode rightson;
	    public TreeNode(String value){
            this.value.append(value);
	        this.leftson = null;
	        this.rightson = null;
	    }
	}
   public static TreeNode str2tree(String[] str){
	   TreeNode rootNode = new TreeNode(str[0]);
	   Queue<TreeNode> nodeQue = new LinkedList<TreeNode>();
	   nodeQue.add(rootNode);
	   while(i<str.length){
		   TreeNode tmp = nodeQue.remove();
		   tmp.leftson = new TreeNode(str[i++]);
		   nodeQue.add(tmp.leftson);
		   if(i<str.length){
		   tmp.rightson = new TreeNode(str[i++]);
		   nodeQue.add(tmp.rightson);
		   }
	   }
	   return rootNode;
   }

   public static String tree2str(TreeNode rootNode){
	   Queue<TreeNode> que = new LinkedList<TreeNode>();
	   String str = "";
	   que.add(rootNode);
	   while(que.isEmpty()==false)
		   {
		   	TreeNode tmpNode = que.remove();
		   	str = str+tmpNode.value[0];
		   	if(tmpNode.leftson !=null)
		   	que.add(tmpNode.leftson);
		   	if(tmpNode.rightson !=null)
		   	que.add(tmpNode.rightson);
		   }
	   return str;
   }
   public static TreeNode copyTree(TreeNode rootNode){
	   if(rootNode==null){
		   return rootNode;
	   }
	   else{
		   TreeNode newRoot = new TreeNode(rootNode.value);
		   TreeNode leftSon = copyTree(rootNode.leftson);
		   TreeNode rightSon = copyTree(rootNode.rightson);
		   newRoot.leftSon = leftSon;
		   newRoot.rightSon = rightSon;
		   return newRoot;
	   }
   }
	   
   public static void main(String[] args){
		String[] str = new String[]{"a","b","c","d","e","f","g","h","i"};
		TreeNode rootNode = str2tree(str);
		String str1 = tree2str(rootNode);
		System.out.println(str1);
		
	} 
}