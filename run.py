import os
import argparse


# Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 4  # e.g. 4 gpus
MASTER_ADDR = '127.0.0.1'  # e.g. '114.213.214.197' for multi nodes
MASTER_PORT = 3006  # e.g. 0~65536
NODE_RANK = 0  # e.g. 0/1/2

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_dist_launch(args):  # some examples
    # use multi nodes, e.g. 2
    # for node 0
    if args.dist == 'm0':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:} ".format(
            NPROC_PER_NODE, 2, 0, MASTER_ADDR, MASTER_PORT)
    # for node 0
    elif args.dist == 'm1':
        return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
               "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:} ".format(
            NPROC_PER_NODE, 2, 1, MASTER_ADDR, MASTER_PORT)

    # use 1 node, 4 gpus
    elif args.dist == 'f4':
        return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.run --nproc_per_node=4 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    # use one gpu, e.g. 'gpu0'
    elif args.dist.startswith('gpu'):
        num = int(args.dist[3:])
        assert 0 <= num <= 3
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.run --nproc_per_node=1 " \
               "--nnodes={:} --node_rank={:} " \
               "--master_addr={:} --master_port={:} ".format(num, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    else:
        raise ValueError


def run(args):
    if args.task in ['sda', 'cuhk', 'icfg', 'rstp']:
        args.config = 'configs/' + args.task + '.yaml'
        print(args.task, args.config)

        dist_launch = get_dist_launch(args)
        os.system(f"{dist_launch} "
                  f"Retrieval.py --config {args.config} "
                  f"--task {args.task} --output_dir {args.output_dir} --bs {args.bs} --epo {args.epo} "
                  f"--re {args.re} --lr {args.lr} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")
    else:
        raise NotImplementedError(f"task == {args.task}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='sda', type=str)
    parser.add_argument('--dist', default='f4', type=str, help="see func get_dist_launch for details")
    parser.add_argument('--output_dir', default='out/pre_sda', type=str, help='local path')
    parser.add_argument('--checkpoint', default='data/checkpoint/pretrain.pth', type=str, help="for fine-tuning")

    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--region', default=0.0, type=float, help="region loss weight")

    args = parser.parse_args()

    run(args)
