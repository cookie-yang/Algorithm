public class randomlinkedList{
    public class RandomListNode {
        int label;
        RandomListNode next, random;

        RandomListNode(int x) {
            this.label = x;
        }
    };
    public static RandomListNode deepcopy(RandomListNode root){
        if((head==null)||(head.next==null&&head.random==null))return head;
		 Map<RandomListNode,RandomListNode>nodeMap = new HashMap<RandomListNode,RandomListNode>();
		 RandomListNode node = new RandomListNode(0);
		 RandomListNode tmp = head;
		 RandomListNode tmp1 = node;
		 while(tmp !=null){
			 tmp1.next = new RandomListNode(tmp.label);
			 nodeMap.put(tmp,tmp1.next);
			 tmp1 = tmp1.next;
			 tmp1.random = tmp.random;
			 tmp = tmp.next;
		 }
		 tmp = head;
		 while(tmp1.next.random !=null){
			 tmp1.next.random = nodeMap.get(tmp1.next.random);
			 tmp1 = tmp1.next;
		 }
		 return node.next;
    }

    public static void main(String[] args) {
        
    }
}