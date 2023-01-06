def train(data, dev_data, test_data, labeler, vocab, config, bert, language_embedder):
    optimizer_label = torch.optim.AdamW(filter(lambda p: p.requires_grad, labeler.model.parameters()), lr=config.learning_rate, \
                                                betas=(config.beta_1, config.beta_2), eps=config.epsilon)  
    decay, decay_step = config.decay, config.decay_steps
    l = lambda epoch: decay ** (epoch // decay_step)  
    scheduler_label = torch.optim.lr_scheduler.LambdaLR(optimizer_label, lr_lambda=l)

    optimizer_lang = torch.optim.AdamW(filter(lambda p: p.requires_grad, language_embedder.parameters()), lr=config.learning_rate, \
                                                betas=(config.beta_1, config.beta_2), eps=config.epsilon)
    scheduler_lang = torch.optim.lr_scheduler.LambdaLR(optimizer_lang, lr_lambda=l)

    optimizer_bert = torch.optim.AdamW(filter(lambda p: p.requires_grad, bert.parameters()), lr=5e-6, eps=1e-8)
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.train_epochs * batch_num)

    for epoch in range(config.train_epochs):
        language_embedder.train()
        labeler.model.train()
        bert.train()

        for onebatch in data_iter(data, config.train_batch_size, True):
            # Forward pass
            lang_embedding = language_embedder(lang_ids)
            pgnbert_hidden = bert(input_ids=bert_indices_tensor, token_type_ids=bert_segments_tensor, 
                                        bert_pieces=bert_pieces_tensor, lang_embedding=lang_embedding)
            labeler.forward(words, extwords, predicts, inmasks, pgnbert_hidden)

            # Compute loss
            loss, stat = labeler.compute_loss(labels, outmasks)
            loss = loss / config.update_every

            # ZERO_GRAD
            optimizer_lang.zero_grad()
            optimizer_bert.zero_grad()
            optimizer_label.zero_grad()

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                                max_norm=config.clip)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, bert.parameters()), \
                                                max_norm=config.clip)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, language_embedder.parameters()), \
                                                max_norm=config.clip)

            # Update parameters after each iteration
            optimizer_lang.step()
            optimizer_bert.step()
            optimizer_label.step()

        # Schduler step after each epoch
        scheduler_lang.step()
        scheduler_bert.step()
        scheduler_label.step()