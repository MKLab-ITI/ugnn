import torch


def training(
    model,
    optimizer,
    train,
    valid,
    test,
    epochs=10000,
    patience=100,
    verbose=None,
    clip=None,
    assess_minimum=False,
):
    best_valid_loss = float("inf")
    best_train_loss = float("inf")
    best_state = None
    max_patience = patience
    patience = 0
    model.eval()
    for epoch in range(epochs):
        # training
        model.train()
        optimizer.zero_grad()
        loss = train.loss(model)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        # compute losses
        model.eval()
        with torch.no_grad():
            train_loss = train.loss(model)
            valid_loss = valid.loss(model)
        # check for patience
        if valid_loss < best_valid_loss:
            if verbose is not None:
                with torch.no_grad():
                    train_acc = train.evaluate(model)
                    valid_acc = valid.evaluate(model)
                    test_acc = test.evaluate(model)
                    best_test_loss = test.loss(model)
            best_valid_loss = valid_loss
            patience = max_patience
            # best_state = model.state_dict()
            if assess_minimum:
                # with torch.no_grad():
                #    test_loss = train.loss(model)
                """brittle = [float(test_loss.cpu())]
                noise = 1.E-2
                for _ in range(10):
                    test_loss2 = test.loss(model, noise=noise)
                    brittle.append(float(test_loss2.cpu()))
                import numpy as np
                brittle = np.std(brittle) / noise"""
                # out = model(train)
                # brittle = torch.mean((out-first_out)**2)**0.5
        elif train_loss < best_train_loss:
            best_train_loss = train_loss
            patience = max_patience
        patience -= 1
        # print current status
        if verbose is not None:
            print(
                f"\r{verbose} Epoch {epoch}"
                f"\tloss {valid_loss:.3f} (best {best_valid_loss:.3f})"
                f"\tbest test loss {best_test_loss:.3f}"
                f"\ttrain acc {train_acc:.3f}"
                f"\tvalid acc {valid_acc:.3f}"
                f"\ttest acc {test_acc:.3f}",
                # f"\tbrittle {brittle if assess_minimum else '---'}",
                end="",
            )
        # stop if running out of patience
        if patience < 0:
            break
    # model.load_state_dict(best_state)
    return test_acc, best_valid_loss
    # return best_test_loss
