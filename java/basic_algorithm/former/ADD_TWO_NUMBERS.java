public class ADD_TWO_NUMBERS{
    public class ListNode{
        int val;
        ListNode next;
        ListNode(int x){val = x;}
    }
    
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int tmp = 0;
        ListNode p = l1;
        ListNode q = l2;
        ListNode l3 =new ListNode(0);
        ListNode head = l3;
        while(p !=null || q!=null){
            int x = (p !=null)?p.val:0;
            int y = (q !=null)?q.val:0;
            int sum = (x+y+tmp);
            l3.next = new ListNode(sum%10);
            tmp = sum/10;
            if(p !=null) p=p.next;
            if(q !=null) q=q.next;
            l3 = l3.next;
        }
        if(tmp ==1) l3.next = new ListNode(1);
        return head.next;
}
    public static void main(String[] args) {
        
    }
}