import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from utils.arguments import load_opt_command

def main(args=None):
    opt, cmdline_args = load_opt_command(args)

    from trainer import MoAI_Trainer as Trainer
    trainer = Trainer(opt)
    
    if cmdline_args.command == 'train':
        trainer.train()
    elif cmdline_args.command == 'eval':
        trainer.opt['LLM']['GRAD_CKPT']=False
        trainer.eval()

if __name__ == "__main__":
    main()